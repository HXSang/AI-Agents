"""External API Service Implementation"""

import json
import time
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

import httpx
from config import settings
from models.models import (
    BalanceScoreDTO,
    DailySnapshot,
    EnergyHealthSummary,
    ExternalAPIOnboardingPayload,
    HealthSummary,
    HRHealthSummary,
    SleepHealthSummary,
    StepsHealthSummary,
)
from services.external_api_service import IExternalAPIService
from utils.logger import logger


class ExternalAPIServiceImpl(IExternalAPIService):
    """Implementation of external API service"""

    def __init__(self):
        self.base_url = settings.be_app_integrate_host
        self.email = settings.be_app_email
        self.password = settings.be_app_password
        self._bearer_token: Optional[str] = None
        self._token_expires_at: Optional[float] = None  # Unix timestamp
        self._is_refreshing_token: bool = False  # Flag to prevent concurrent login
        self._timeout = 30.0
        # Default token expiration time (1 hour) if not provided by API
        self._default_token_expiration_seconds = 3600

    # ─────────────────────────────────────────────────────────────
    # HELPER METHODS
    # ─────────────────────────────────────────────────────────────

    def _is_token_valid(self) -> bool:
        """Check if current token is valid and not expired

        Returns:
            True if token exists and is not expired, False otherwise
        """
        if not self._bearer_token:
            return False

        if self._token_expires_at is None:
            # If expiration time is not set, assume token is invalid
            return False

        # Check if token is expired (with 5 minute buffer)
        buffer_seconds = 300  # 5 minutes
        current_time = time.time()
        if current_time >= (self._token_expires_at - buffer_seconds):
            logger.debug(
                f"🔑 Token expired or expiring soon. Current: {current_time}, Expires: {self._token_expires_at}"
            )
            return False

        return True

    async def _ensure_authenticated(self) -> bool:
        """Ensure bearer token is available and base_url is configured

        Returns:
            True if authenticated and configured, False otherwise
        """
        # Check if token is valid before attempting login
        if not self._is_token_valid():
            token = await self.login_and_get_token()
            if not token:
                logger.error("❌ Failed to get bearer token")
                return False
        else:
            logger.debug("✅ Using existing valid token")

        if not self.base_url:
            logger.error("❌ External API host not configured")
            return False

        return True

    def _get_headers(self) -> Dict[str, str]:
        """Get standard headers with bearer token

        Returns:
            Dict with Authorization and Content-Type headers
        """
        return {
            "Authorization": f"Bearer {self._bearer_token}",
            "Content-Type": "application/json",
        }

    async def _make_request(
        self,
        method: str,
        url: str,
        headers: Optional[Dict[str, str]] = None,
        json_data: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
        success_status_codes: List[int] = [200],
    ) -> Optional[httpx.Response]:
        """Make HTTP request with standard error handling

        Args:
            method: HTTP method (GET, POST, PUT, DELETE)
            url: Request URL
            headers: Optional headers (defaults to standard headers with bearer token)
            json_data: Optional JSON payload for POST/PUT
            params: Optional query parameters
            success_status_codes: List of success status codes (default: [200])

        Returns:
            Response object if successful, None otherwise
        """
        try:
            if headers is None:
                headers = self._get_headers()

            async with httpx.AsyncClient(timeout=self._timeout) as client:
                if method.upper() == "GET":
                    response = await client.get(url, headers=headers, params=params)
                elif method.upper() == "POST":
                    response = await client.post(
                        url, json=json_data, headers=headers, params=params
                    )
                elif method.upper() == "PUT":
                    response = await client.put(
                        url, json=json_data, headers=headers, params=params
                    )
                elif method.upper() == "DELETE":
                    response = await client.delete(url, headers=headers, params=params)
                else:
                    logger.error(f"❌ Unsupported HTTP method: {method}")
                    return None

                if response.status_code in success_status_codes:
                    return response
                else:
                    logger.error(
                        f"❌ Request failed. Status: {response.status_code}, Response: {response.text}"
                    )
                    return None

        except Exception as e:
            logger.error(f"❌ Error making request: {str(e)}")
            return None

    async def login_and_get_token(self) -> Optional[str]:
        """Login to external API and get bearer token.

        This method checks if the current token is valid before attempting a new login.
        Only logs in if token is missing or expired.

        Returns:
            Bearer token string if successful, None otherwise
        """
        # Check if token is still valid
        if self._is_token_valid():
            logger.debug("✅ Token is still valid, reusing existing token")
            return self._bearer_token

        # Prevent concurrent login attempts
        if self._is_refreshing_token:
            logger.debug("⏳ Token refresh already in progress, waiting...")
            # Wait a bit and check again
            import asyncio

            await asyncio.sleep(0.5)
            if self._is_token_valid():
                return self._bearer_token
            # If still invalid, proceed with login (might be a race condition)

        try:
            if not self.base_url or not self.email or not self.password:
                logger.warning("External API credentials not configured")
                return None

            # Set flag to prevent concurrent logins
            self._is_refreshing_token = True

            login_url = f"{self.base_url}/api/auth/login"
            login_data = {"email": self.email, "password": self.password}

            logger.info(f"🔐 Logging in to external API: {login_url}")

            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(login_url, json=login_data)

                if response.status_code == 200:
                    response_data = response.json()

                    # Handle nested response structure: {"data": {"token": "xxx"}}
                    data = None
                    if "data" in response_data and isinstance(
                        response_data["data"], dict
                    ):
                        data = response_data["data"]
                        self._bearer_token = data.get("token") or data.get(
                            "access_token"
                        )
                    else:
                        # Fallback to direct token access
                        data = response_data
                        self._bearer_token = response_data.get(
                            "token"
                        ) or response_data.get("access_token")

                    if self._bearer_token:
                        # Extract expiration time from response
                        expires_in = None
                        if data:
                            expires_in = data.get("expiresIn")  # seconds

                        # Calculate expiration timestamp
                        if expires_in:
                            self._token_expires_at = time.time() + expires_in
                            logger.info(
                                f"✅ Token expires in {expires_in} seconds (at timestamp {self._token_expires_at})"
                            )
                        else:
                            # Use default expiration if not provided
                            self._token_expires_at = (
                                time.time() + self._default_token_expiration_seconds
                            )
                            logger.info(
                                f"✅ Token expiration not provided, using default: {self._default_token_expiration_seconds} seconds"
                            )

                        logger.info("✅ Successfully logged in to external API")
                        logger.info(f"🔑 Token: {self._bearer_token[:20]}...")
                        return self._bearer_token
                    else:
                        logger.error("❌ No token found in login response")
                        logger.error(f"📄 Response structure: {response_data}")
                        self._token_expires_at = None
                        return None
                else:
                    logger.error(
                        f"❌ Login failed with status {response.status_code}: {response.text}"
                    )
                    self._token_expires_at = None
                    return None

        except Exception as e:
            logger.error(f"❌ Error during login: {str(e)}")
            self._token_expires_at = None
            return None

    async def save_onboarding_data(self, user_data: Dict[str, Any]) -> bool:
        """Save onboarding data to external API"""
        try:
            if not await self._ensure_authenticated():
                return False

            # Create payload using model
            payload_model = ExternalAPIOnboardingPayload.from_user_data(user_data)

            # Convert to dict and remove None values
            payload = payload_model.model_dump(exclude_none=True)

            save_url = f"{self.base_url}/api/onboarding/process"

            logger.info(f"💾 Saving onboarding data to external API: {save_url}")
            logger.info(f"📊 Payload: {json.dumps(payload, indent=2)}")

            response = await self._make_request(
                method="POST",
                url=save_url,
                json_data=payload,
                success_status_codes=[200, 201],
            )

            if response:
                logger.info("✅ Successfully saved onboarding data to external API")
                return True
            else:
                return False

        except Exception as e:
            logger.error(f"❌ Error saving onboarding data: {str(e)}")
            return False

    async def get_user_info(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get user info (name, email) from /api/users/{user_id}"""
        try:
            if not await self._ensure_authenticated():
                return None

            # Call API to get user info
            get_url = f"{self.base_url}/api/users/{user_id}"

            logger.info(f"📥 Fetching user info from DB: {get_url}")

            response = await self._make_request(
                method="GET",
                url=get_url,
                success_status_codes=[200, 404],
            )

            if response:
                if response.status_code == 200:
                    user_info = response.json()
                    if isinstance(user_info, dict):
                        logger.info(
                            f"✅ Successfully fetched user info for user {user_id}"
                        )
                        return user_info
                    return None
                elif response.status_code == 404:
                    logger.info(f"ℹ️ No user info found for user {user_id}")
                    return None

            return None

        except Exception as e:
            logger.error(f"❌ Error getting user info: {str(e)}")
            return None

    async def get_user_data(self, user_id: str) -> Optional[Dict[str, Any]]:
        try:
            if not await self._ensure_authenticated():
                return None

            # Call API to get user data from DB
            get_url = f"{self.base_url}/api/onboarding/users/{user_id}/profiles"

            logger.info(f"📥 Fetching user data from DB: {get_url}")

            response = await self._make_request(
                method="GET",
                url=get_url,
                success_status_codes=[200, 404],
            )

            user_data: Optional[Dict[str, Any]] = None
            if response:
                if response.status_code == 200:
                    # ── Parse body safely (was: bare response.json()) ─────
                    try:
                        response_data = response.json()
                    except Exception as json_err:
                        logger.warning(
                            f"⚠️ user_profile response not JSON for "
                            f"{user_id} ({get_url}): {type(json_err).__name__}: "
                            f"{json_err}"
                        )
                        response_data = None

                    if response_data is not None:
                        if isinstance(response_data, list):
                            if response_data:
                                candidate = response_data[0]
                                dump = getattr(candidate, "model_dump", None)
                                if callable(dump):
                                    try:
                                        user_data = dump()
                                    except Exception as dump_err:
                                        logger.warning(
                                            f"⚠️ user_profile[0].model_dump() "
                                            f"failed for {user_id}: "
                                            f"{type(dump_err).__name__}: {dump_err}"
                                        )
                                        user_data = (
                                            candidate
                                            if isinstance(candidate, dict)
                                            else None
                                        )
                                elif isinstance(candidate, dict):
                                    user_data = candidate
                                else:
                                    logger.warning(
                                        f"⚠️ user_profile[0] is neither dict "
                                        f"nor Pydantic model (type={type(candidate).__name__})"
                                    )
                                    user_data = None
                            else:
                                logger.info(
                                    f"ℹ️ user_profile list is empty for {user_id}"
                                )
                                user_data = None
                        elif isinstance(response_data, dict) and "data" in response_data:
                            data_field = response_data["data"]
                            if isinstance(data_field, dict):
                                user_data = data_field
                            else:
                                # Pydantic model inside {"data": ...}
                                dump = getattr(data_field, "model_dump", None)
                                if callable(dump):
                                    try:
                                        user_data = dump()
                                    except Exception as dump_err:
                                        logger.warning(
                                            f"⚠️ user_profile.data.model_dump() "
                                            f"failed for {user_id}: "
                                            f"{type(dump_err).__name__}: {dump_err}"
                                        )
                                        user_data = None
                                else:
                                    logger.warning(
                                        f"⚠️ user_profile.data has unexpected "
                                        f"type: {type(data_field).__name__}"
                                    )
                                    user_data = None
                        else:
                            logger.warning(
                                f"⚠️ Unexpected user_profile response "
                                f"structure for {user_id}: keys="
                                f"{list(response_data.keys()) if isinstance(response_data, dict) else type(response_data).__name__}"
                            )
                elif response.status_code == 404:
                    logger.info(f"ℹ️ No user data found for user {user_id}")
                try:
                    user_info = await self.get_user_info(user_id)
                except Exception as info_err:
                    logger.warning(
                        f"⚠️ get_user_info failed for {user_id}: "
                        f"{type(info_err).__name__}: {info_err}"
                    )
                    user_info = None

                # Normalize: ensure we have a dict before merging fields.
                if user_data and not isinstance(user_data, dict):
                    logger.warning(
                        f"⚠️ user_data for {user_id} is not a dict "
                        f"(type={type(user_data).__name__}); coercing to empty dict"
                    )
                    user_data = {}
                if user_data is None:
                    user_data = {}

                if user_info and isinstance(user_info, dict):
                    first_name = user_info.get("firstName") or ""
                    last_name = user_info.get("lastName") or ""
                    if first_name or last_name:
                        name_parts = [
                            str(part) for part in [first_name, last_name] if part
                        ]
                        if name_parts:
                            user_data["name"] = " ".join(name_parts)

                    # Add email — same guard pattern.
                    email = user_info.get("email")
                    if email:
                        user_data["email"] = email

                    logger.info(
                        f"✅ Merged user info (name, email) into user data "
                        f"for user {user_id}"
                    )

                if user_data:
                    logger.info(
                        f"✅ Successfully fetched user data for user {user_id}"
                    )
                    return user_data

                # If no user_data but have user_info, return at least name and email.
                if user_info and isinstance(user_info, dict):
                    result: Dict[str, Any] = {}
                    first_name = user_info.get("firstName") or ""
                    last_name = user_info.get("lastName") or ""
                    if first_name or last_name:
                        name_parts = [
                            str(part) for part in [first_name, last_name] if part
                        ]
                        if name_parts:
                            result["name"] = " ".join(name_parts)
                    email = user_info.get("email")
                    if email:
                        result["email"] = email
                    return result if result else None

                return None

        except Exception as e:
            logger.error(
                f"❌ Error getting user data for {user_id}: "
                f"{type(e).__name__}: {e}"
            )
            return None

    async def get_profile_id(self, user_id: str) -> Optional[str]:
        """Get profile_id for a user from cache or API

        Args:
            user_id: User ID

        Returns:
            profile_id if found, None otherwise
        """
        try:
            # First try to get from user_data API
            user_data = await self.get_user_data(user_id)
            if user_data:
                # Extract profile_id from user_data
                if isinstance(user_data, dict):
                    profile_id = user_data.get("profile_id")
                    if profile_id:
                        logger.info(
                            f"✅ Found profile_id for user {user_id}: {profile_id}"
                        )
                        return profile_id
                elif hasattr(user_data, "profile_id"):
                    profile_id = user_data.profile_id
                    if profile_id:
                        logger.info(
                            f"✅ Found profile_id for user {user_id}: {profile_id}"
                        )
                        return profile_id
                elif hasattr(user_data, "model_dump"):
                    user_data_dict = user_data.model_dump()
                    profile_id = user_data_dict.get("profile_id")
                    if profile_id:
                        logger.info(
                            f"✅ Found profile_id for user {user_id}: {profile_id}"
                        )
                        return profile_id

            logger.warning(f"⚠️ profile_id not found for user {user_id}")
            return None

        except Exception as e:
            logger.error(f"❌ Error getting profile_id: {str(e)}")
            return None

    async def get_calendar_events(
        self,
        user_id: str,
        provider_name: Optional[str] = "",
        timezone: Optional[str] = None,
        start_date_time: Optional[str] = None,
        end_date_time: Optional[str] = None,
        include_cancelled: Optional[bool] = False,
        include_declined: Optional[bool] = False,
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Get calendar events from external API within date range.

        Args:
            user_id: User ID
            provider_name: Optional provider name (default: empty string)
            timezone: Optional IANA timezone
            start_date_time: Optional start datetime (format: "YYYY-MM-DDTHH:mm")
            end_date_time: Optional end datetime (format: "YYYY-MM-DDTHH:mm")
            include_cancelled: Include cancelled events (default: False)
            include_declined: Include declined events (default: False)

        Returns:
            List of calendar events or None if error
        """
        try:
            if not await self._ensure_authenticated():
                return None

            # Build query params
            query_params = {}
            if provider_name:
                query_params["providerName"] = provider_name
            if user_id:
                query_params["userId"] = user_id
            if timezone:
                query_params["timezone"] = timezone
            if start_date_time:
                query_params["startDateTime"] = start_date_time
            if end_date_time:
                query_params["endDateTime"] = end_date_time
            if include_cancelled is not None:
                query_params["includeCancelled"] = str(include_cancelled).lower()
            if include_declined is not None:
                query_params["includeDeclined"] = str(include_declined).lower()

            # Use /calendar/events endpoint (not /today)
            url = f"{self.base_url}/api/calendar/events"

            logger.info(f"📅 Fetching calendar events from: {url}")
            logger.info(f"📋 Query params: {query_params}")

            response = await self._make_request(
                method="GET",
                url=url,
                params=query_params,
                success_status_codes=[200, 404],
            )

            if response:
                if response.status_code == 200:
                    events = response.json()
                    if isinstance(events, list):
                        logger.info(
                            f"✅ Successfully fetched {len(events)} calendar events for user {user_id}"
                        )
                        return events
                    else:
                        logger.warning(f"⚠️ Unexpected response format: {type(events)}")
                        return []
                elif response.status_code == 404:
                    logger.info(f"ℹ️ No calendar events found for user {user_id}")
                    return []

            return None

        except Exception as e:
            logger.error(f"❌ Error getting calendar events: {str(e)}")
            return None

    async def get_reminders(
        self,
        user_id: str,
        start_date: str,
        end_date: str,
        timezone: Optional[str] = None,
        provider_name: Optional[str] = None,
        incomplete_only: bool = False,
        page: int = 1,
        size: int = 50,
    ) -> Optional[List[Dict[str, Any]]]:
        extra_params: Dict[str, Any] = {"page": page, "size": size}
        if provider_name:
            extra_params["providerName"] = provider_name
        extra_params["incompleteOnly"] = incomplete_only

        return await self._fetch_calendar_list(
            path_suffix="reminders",
            user_id=user_id,
            start_date=start_date,
            end_date=end_date,
            timezone=timezone,
            extra_params=extra_params,
            entity_label="reminders",
            response_kind="page",
        )

    async def get_work_hours(
        self,
        user_id: str,
        date: str,
        timezone: Optional[str] = None,
        provider_name: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        try:
            if not await self._ensure_authenticated():
                return None

            params: Dict[str, Any] = {"userId": user_id, "date": date}
            if timezone:
                params["timezone"] = timezone
            if provider_name:
                params["providerName"] = provider_name

            url = f"{self.base_url}/api/calendar/work-hours"
            logger.info(f"Fetching work hours from: {url}")
            response = await self._make_request("GET", url, params=params)

            if not response:
                return None
            data = response.json()
            return data if isinstance(data, dict) else None
        except Exception as e:
            logger.error(f"❌ Error getting work hours: {str(e)}")
            return None

    async def get_reminder_lists(
        self, user_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        try:
            if not await self._ensure_authenticated():
                return []

            params: Dict[str, Any] = {}
            if user_id:
                params["userId"] = user_id

            url = f"{self.base_url}/api/calendar/reminder-lists"
            logger.info(f"Fetching reminder lists from: {url}")
            response = await self._make_request("GET", url, params=params)

            if not response:
                return []
            data = response.json()
            return data if isinstance(data, list) else []
        except Exception as e:
            logger.error(f"❌ Error getting reminder lists: {str(e)}")
            return []

    async def get_reminder_by_id(
        self,
        reminder_id: str,
        user_id: Optional[str] = None,
        original_due_date: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        try:
            if not await self._ensure_authenticated():
                return None

            params: Dict[str, Any] = {}
            if user_id:
                params["userId"] = user_id
            if original_due_date:
                params["originalDueDate"] = original_due_date

            url = f"{self.base_url}/api/calendar/reminders/{reminder_id}"
            logger.info(f"Fetching reminder {reminder_id} from: {url}")
            response = await self._make_request("GET", url, params=params)

            if not response:
                return None
            data = response.json()
            return data if isinstance(data, dict) else None
        except Exception as e:
            logger.error(f"❌ Error getting reminder by id: {str(e)}")
            return None

    async def get_calendar_event_by_id(
        self,
        event_id: str,
        user_id: Optional[str] = None,
        original_start_time: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        try:
            if not await self._ensure_authenticated():
                return None

            params: Dict[str, Any] = {}
            if user_id:
                params["userId"] = user_id
            if original_start_time:
                params["originalStartTime"] = original_start_time

            url = f"{self.base_url}/api/calendar/events/{event_id}"
            logger.info(f"Fetching event {event_id} from: {url}")
            response = await self._make_request("GET", url, params=params)

            if not response:
                return None
            data = response.json()
            return data if isinstance(data, dict) else None
        except Exception as e:
            logger.error(f"❌ Error getting calendar event by id: {str(e)}")
            return None

    async def get_calendar_events_today(
        self,
        user_id: Optional[str] = None,
        profile_id: Optional[str] = None,
        provider_name: Optional[str] = None,
        timezone: Optional[str] = None,
        include_cancelled: Optional[bool] = False,
        include_declined: Optional[bool] = False,
    ) -> List[Dict[str, Any]]:
        try:
            if not await self._ensure_authenticated():
                return []

            params: Dict[str, Any] = {}
            if user_id:
                params["userId"] = user_id
            if profile_id:
                params["profileId"] = profile_id
            if provider_name:
                params["providerName"] = provider_name
            if timezone:
                params["timezone"] = timezone
            if include_cancelled:
                params["includeCancelled"] = include_cancelled
            if include_declined:
                params["includeDeclined"] = include_declined

            url = f"{self.base_url}/api/calendar/events/today"
            logger.info(f"Fetching today's events from: {url}")
            response = await self._make_request("GET", url, params=params)

            if not response:
                return []
            data = response.json()
            return data if isinstance(data, list) else []
        except Exception as e:
            logger.error(f"❌ Error getting today's events: {str(e)}")
            return []

    async def get_calendar_sync_status(
        self,
        user_id: Optional[str] = None,
        profile_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        try:
            if not await self._ensure_authenticated():
                return []

            params: Dict[str, Any] = {}
            if user_id:
                params["userId"] = user_id
            if profile_id:
                params["profileId"] = profile_id

            url = f"{self.base_url}/api/calendar/sync/status"
            logger.info(f"Fetching calendar sync status from: {url}")
            response = await self._make_request("GET", url, params=params)

            if not response:
                return []
            data = response.json()
            return data if isinstance(data, list) else []
        except Exception as e:
            logger.error(f"❌ Error getting calendar sync status: {str(e)}")
            return []

    async def _fetch_calendar_list(
        self,
        *,
        path_suffix: str,
        user_id: str,
        start_date: str,
        end_date: str,
        timezone: Optional[str],
        extra_params: Optional[Dict[str, Any]],
        entity_label: str,
        response_kind: str,
    ) -> Optional[List[Dict[str, Any]]]:
        try:
            if not await self._ensure_authenticated():
                return None

            params: Dict[str, Any] = {
                "userId": user_id,
                "startDate": start_date,
                "endDate": end_date,
            }
            if timezone:
                params["timezone"] = timezone
            if extra_params:
                params.update(extra_params)

            url = f"{self.base_url}/api/calendar/{path_suffix}"
            logger.info(f"Fetching {entity_label} from: {url}")
            logger.info(f"Query params: {params}")

            response = await self._make_request(
                method="GET",
                url=url,
                params=params,
                success_status_codes=[200, 404],
            )

            if not response:
                return None
            if response.status_code == 404:
                logger.info(f"No {entity_label} data for user {user_id}")
                return []

            data = response.json()

            def _extract_content(payload: Any) -> Optional[List[Dict[str, Any]]]:
                if response_kind == "list" and isinstance(payload, list):
                    return payload
                if isinstance(payload, dict):
                    # Try common pagination wrapper keys (BE may rename).
                    for wrapper_key in ("content", "items", "data", "records", "results"):
                        content = payload.get(wrapper_key)
                        if isinstance(content, list):
                            return content
                    if "date" in payload or "day" in payload:
                        return [payload]
                return None

            content = _extract_content(data)
            if content is None:
                if isinstance(data, dict):
                    keys = sorted(data.keys())
                    sample = {
                        k: (
                            type(v).__name__
                            if not isinstance(v, (str, int, float, bool, type(None)))
                            else (str(v)[:60] if isinstance(v, str) else v)
                        )
                        for k, v in data.items()
                    }
                    logger.warning(
                        f"⚠️ Unexpected {entity_label} response format: "
                        f"dict with keys={keys} sample_types={sample}"
                    )
                else:
                    preview = str(data)[:200]
                    logger.warning(
                        f"⚠️ Unexpected {entity_label} response format: "
                        f"type={type(data).__name__} preview={preview}"
                    )
                return []

            logger.info(
                f"Successfully fetched {len(content)} {entity_label} for user {user_id}"
            )
            return content

        except Exception as e:
            logger.error(f"❌ Error getting {entity_label}: {str(e)}")
            return None

    async def save_conversation(self, user_id: str, payload: Dict[str, Any]) -> bool:
        """Save conversation to database via external API

        Args:
            user_id: User ID
            payload: Conversation payload with profileId, timestamp, userMessage, botResponse, flow

        Returns:
            True if saved successfully, False otherwise
        """
        try:
            if not await self._ensure_authenticated():
                return False

            # Build URL with userId as query param
            save_url = f"{self.base_url}/api/conversations"
            query_params = {"userId": user_id}

            logger.info(f"💾 Saving conversation to external API: {save_url}")
            logger.info(f"📋 Query params: {query_params}")
            response = await self._make_request(
                method="POST",
                url=save_url,
                json_data=payload,
                params=query_params,
                success_status_codes=[200, 201],
            )

            if response:
                return True
            else:
                return False

        except Exception as e:
            logger.error(f"❌ Error saving conversation: {str(e)}")
            return False

    async def get_user_profiles(self, user_id: str) -> Optional[List[Dict[str, Any]]]:
        """Get user profiles from external API (returns list)"""
        try:
            token = await self.login_and_get_token()
            if not token or not self.base_url:
                return None

            url = f"{self.base_url}/api/onboarding/users/{user_id}/profiles"
            headers = {"Authorization": f"Bearer {self._bearer_token}"}

            logger.info(f"👤 Fetching user profiles from: {url}")

            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(url, headers=headers)

                if response.status_code == 200:
                    data = response.json()
                    logger.info(f"✅ Successfully fetched profiles for user {user_id}")
                    # Ensure we return a list
                    return data if isinstance(data, list) else []
                else:
                    logger.warning(
                        f"⚠️ Failed to fetch profiles. Status: {response.status_code}"
                    )
                    return None

        except Exception as e:
            logger.error(f"❌ Error getting user profiles: {str(e)}")
            return None

    async def get_productivity_summary(
        self,
        user_id: str,
        date: Optional[str] = None,
        timezone: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        try:
            if not await self._ensure_authenticated():
                return None

            url = f"{self.base_url}/api/productivity/summary"
            params: Dict[str, Any] = {"userId": user_id}
            if date:
                params["date"] = date
            if timezone:
                params["timezone"] = timezone

            logger.info(f"📊 Fetching productivity summary from: {url}")
            response = await self._make_request("GET", url, params=params)
            if not response:
                return None
            data = response.json()
            return data if isinstance(data, dict) else None
        except Exception as e:
            logger.error(f"❌ Error getting productivity summary: {str(e)}")
            return None

    async def get_productivity_summaries_by_range(
        self,
        user_id: str,
        start_date: str,
        end_date: str,
        timezone: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Get daily productivity summaries by range from /api/productivity/summaries."""
        try:
            if not await self._ensure_authenticated():
                return []
            url = f"{self.base_url}/api/productivity/summaries"
            params: Dict[str, Any] = {
                "userId": user_id,
                "startDate": start_date,
                "endDate": end_date,
            }
            if timezone:
                params["timezone"] = timezone
            response = await self._make_request("GET", url, params=params)
            if not response:
                return []
            data = response.json()
            if not isinstance(data, list):
                return []
            return [item for item in data if isinstance(item, dict)]
        except Exception as e:
            logger.error(f"❌ Error getting productivity summaries by range: {str(e)}")
            return []

    async def sync_emails(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Trigger email sync for user via POST /emails/sync"""
        try:
            token = await self.login_and_get_token()
            if not token or not self.base_url:
                return None

            url = f"{self.base_url}/api/emails/sync"
            headers = {
                "Authorization": f"Bearer {self._bearer_token}",
                "Content-Type": "application/json",
            }
            payload = {"userId": user_id}

            logger.info(f"🔄 Triggering email sync for user {user_id}")

            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(url, headers=headers, json=payload)

                if response.status_code in [
                    200,
                    201,
                    202,
                ]:  # Accept 202 for async operations
                    data = response.json()
                    logger.info(f"✅ Email sync triggered for user {user_id}")
                    return data
                else:
                    logger.warning(
                        f"⚠️ Failed to trigger email sync. Status: {response.status_code}"
                    )
                    return None

        except Exception as e:
            logger.error(f"❌ Error triggering email sync: {str(e)}")
            return None

    async def check_email_sync_status(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Check email sync status via GET /emails/sync/status"""
        try:
            token = await self.login_and_get_token()
            if not token or not self.base_url:
                return None

            url = f"{self.base_url}/api/emails/sync/status"
            headers = {"Authorization": f"Bearer {self._bearer_token}"}
            params = {"userId": user_id}

            logger.info(f"🔍 Checking email sync status for user {user_id}")

            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(url, headers=headers, params=params)

                if response.status_code == 200:
                    data = response.json()
                    logger.info(f"✅ Email sync status: {data}")
                    return data
                else:
                    logger.warning(
                        f"⚠️ Failed to check email sync status. Status: {response.status_code}"
                    )
                    return None

        except Exception as e:
            logger.error(f"❌ Error checking email sync status: {str(e)}")
            return None

    async def get_latest_mood(
        self,
        user_id: str,
        target_date: Optional[str] = None,
        timezone: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        try:
            if not await self._ensure_authenticated():
                return None

            params: Dict[str, Any] = {"userId": user_id}
            if target_date:
                params["targetDate"] = target_date
            if timezone:
                params["timezone"] = timezone

            url = f"{self.base_url}/api/moods/latest"
            logger.info(f"😊 Fetching latest mood from: {url}")

            response = await self._make_request("GET", url, params=params)
            if not response:
                return None

            data = response.json()
            logger.info(f"✅ Successfully fetched mood for user {user_id}")
            return data
        except Exception as e:
            logger.error(f"❌ Error getting latest mood: {str(e)}")
            return None

    async def get_moods(
        self,
        user_id: Optional[str] = None,
        page: int = 1,
        size: int = 20,
        start_datetime: Optional[str] = None,
        end_datetime: Optional[str] = None,
        timezone: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        try:
            if not await self._ensure_authenticated():
                return None

            params: Dict[str, Any] = {
                "page": page,
                "size": size,
            }
            if user_id:
                params["userId"] = user_id
            if start_datetime:
                params["startDateTime"] = start_datetime
            if end_datetime:
                params["endDateTime"] = end_datetime
            if timezone:
                params["timezone"] = timezone

            url = f"{self.base_url}/api/moods"
            logger.info(f"😊 Fetching moods from: {url}")

            response = await self._make_request("GET", url, params=params)
            if not response:
                return None

            data = response.json()
            return data if isinstance(data, dict) else None
        except Exception as e:
            logger.error(f"❌ Error getting moods: {str(e)}")
            return None

    async def get_health_sample_types(
        self, user_id: str
    ) -> Optional[List[Dict[str, Any]]]:
        """Get available health sample types for a user from external API

        Step 1 of dynamic health data collection: Fetch all available health types for user.
        """
        try:
            token = await self.login_and_get_token()
            if not token or not self.base_url:
                return None

            url = f"{self.base_url}/api/health/samples/types"
            headers = {"Authorization": f"Bearer {self._bearer_token}"}
            params = {"userId": user_id}

            logger.info(f"🏥 Fetching health sample types from: {url}")

            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(url, headers=headers, params=params)

                if response.status_code == 200:
                    data = response.json()
                    logger.info(
                        f"✅ Successfully fetched {len(data)} health types for user {user_id}"
                    )
                    return data
                else:
                    logger.warning(
                        f"⚠️ Failed to fetch health sample types. Status: {response.status_code}"
                    )
                    return None

        except Exception as e:
            logger.error(f"❌ Error getting health sample types: {str(e)}")
            return None

    async def get_all_health_stats_dynamic(
        self, user_id: str, timezone: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Get all health stats dynamically for a user

        Two-step process:
        1. Call /api/health/samples/types to get available types for user
        2. For each type, call /api/health/stats/ with yesterday-today date range

        Args:
            user_id: The user ID
            timezone: Optional timezone (e.g., "Asia/Ho_Chi_Minh")

        Returns:
            Dictionary with type codes as keys and stats as values, or None if failed
        """
        try:
            # Step 1: Get available health sample types for this user
            health_types = await self.get_health_sample_types(user_id)
            if not health_types:
                logger.warning(f"⚠️ No health sample types found for user {user_id}")
                return None

            # Calculate date range: yesterday to today
            today = datetime.now().date()
            yesterday = today - timedelta(days=1)
            start_date = yesterday.strftime("%Y-%m-%d")
            end_date = today.strftime("%Y-%m-%d")

            logger.info(
                f"📊 Fetching health stats for {len(health_types)} types from {start_date} to {end_date}"
            )

            # Step 2: For each type, get stats
            all_stats = {}
            token = await self.login_and_get_token()
            if not token or not self.base_url:
                return None

            headers = {"Authorization": f"Bearer {self._bearer_token}"}

            async with httpx.AsyncClient(timeout=30.0) as client:
                for health_type in health_types:
                    type_code = health_type.get("code")
                    type_name = health_type.get("displayName", type_code)

                    if not type_code:
                        continue

                    try:
                        url = f"{self.base_url}/api/health/stats/"
                        params = {
                            "userId": user_id,
                            "type": type_code,
                            "startDate": start_date,
                            "endDate": end_date,
                        }

                        if timezone:
                            params["timezone"] = timezone

                        logger.info(f"  📈 Fetching {type_name} ({type_code}) stats...")

                        response = await client.get(url, headers=headers, params=params)

                        if response.status_code == 200:
                            stats_data = response.json()
                            all_stats[type_code] = stats_data
                            logger.info(f"    ✅ {type_name}: Success")
                        else:
                            logger.warning(
                                f"    ⚠️ {type_name}: Failed (Status: {response.status_code})"
                            )
                            all_stats[type_code] = None

                    except Exception as e:
                        logger.error(f"    ❌ {type_name}: Error - {str(e)}")
                        all_stats[type_code] = None

            logger.info(
                f"✅ Completed fetching health stats for user {user_id}: "
                f"{len([v for v in all_stats.values() if v is not None])}/{len(health_types)} successful"
            )

            return all_stats

        except Exception as e:
            logger.error(f"❌ Error in get_all_health_stats_dynamic: {str(e)}")
            return None

    async def get_health_summaries(
        self,
        user_id: str,
        type_code: Optional[str] = None,
        timezone: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Get health summaries from /api/health/summaries endpoint.

        Calls /api/health/summaries with current week date range and extracts latest data.

        Args:
            user_id: User ID
            type_code: Optional type code (HR, SLEEP, STEPS, ENERGY). If None, returns all 4 types
            timezone: Required IANA timezone for week-window alignment.
                ``None`` is treated as a caller bug; we fall back to UTC and log.

        Returns:
            Dict with all 4 summaries keyed by type, each containing latest weekly summary
        """
        try:
            token = await self.login_and_get_token()
            if not token or not self.base_url:
                logger.warning(
                    "⚠️ Cannot get health summaries: not authenticated or base_url not configured"
                )
                return None

            if timezone:
                tz = ZoneInfo(timezone)
            else:
                tz = ZoneInfo("UTC")
            now = datetime.now(tz)

            end_date = now.strftime("%Y-%m-%d")
            start_date = (now - timedelta(days=7)).strftime("%Y-%m-%d")

            logger.info(
                f"📊 Fetching health summaries for user {user_id} (type: {type_code or 'all'}) "
                f"from {start_date} to {end_date} (timezone: {timezone or 'UTC'})"
            )

            headers = {"Authorization": f"Bearer {self._bearer_token}"}
            url = f"{self.base_url}/api/health/summaries"
            params = {
                "userId": user_id,
                "startDate": start_date,
                "endDate": end_date,
            }
            if timezone:
                params["timezone"] = timezone

            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.get(url, headers=headers, params=params)

                if response.status_code == 200:
                    data = response.json()

                    if not isinstance(data, list):
                        logger.warning(
                            "⚠️ Unexpected response format from health summaries API"
                        )
                        return None

                    # Group by type and get latest (most recent) summary for each type
                    summaries_by_type = {}
                    for item in data:
                        try:
                            item_type = item.get("type", "UNKNOWN")

                            # Parse based on type
                            if item_type == "ENERGY":
                                parsed_item = EnergyHealthSummary.model_validate(item)
                            elif item_type == "HR":
                                parsed_item = HRHealthSummary.model_validate(item)
                            elif item_type == "SLEEP":
                                parsed_item = SleepHealthSummary.model_validate(item)
                            elif item_type == "STEPS":
                                parsed_item = StepsHealthSummary.model_validate(item)
                            else:
                                logger.warning(
                                    f"⚠️ Unknown health summary type: {item_type}, skipping"
                                )
                                continue

                            # Keep the most recent summary for each type (by startDate)
                            if item_type not in summaries_by_type:
                                summaries_by_type[item_type] = parsed_item
                            else:
                                # Compare startDate to keep the latest
                                current_start = summaries_by_type[
                                    item_type
                                ].dateRange.startDate
                                new_start = parsed_item.dateRange.startDate
                                if new_start > current_start:
                                    summaries_by_type[item_type] = parsed_item

                        except Exception as parse_error:
                            logger.warning(
                                f"⚠️ Failed to parse health summary: {str(parse_error)}, skipping"
                            )
                            continue

                    # Filter by type_code if provided
                    if type_code:
                        if type_code in summaries_by_type:
                            result = {type_code: summaries_by_type[type_code]}
                        else:
                            logger.warning(f"⚠️ No {type_code} health summary found")
                            return None
                    else:
                        result = summaries_by_type

                    logger.info(
                        f"✅ Successfully retrieved and parsed {len(result)} health summaries: {', '.join(result.keys())}"
                    )
                    return result if result else None
                else:
                    logger.warning(
                        f"⚠️ Failed to get health summaries: {response.status_code}"
                    )
                    return None

        except Exception as e:
            logger.error(f"❌ Error getting health summaries: {str(e)}")
            return None

    async def get_health_summaries_by_range(
        self,
        user_id: str,
        start_date: str,
        end_date: str,
        timezone: Optional[str] = None,
    ) -> Optional[List["HealthSummary"]]:
        """Get health summaries theo date range từ /api/health/summaries (tất cả types)

        Args:
            user_id: User ID
            start_date: Start date in ISO format (e.g., "2026-01-25T17:00:00Z")
            end_date: End date in ISO format (e.g., "2026-02-01T16:59:59.999Z")
            timezone: Optional timezone

        Returns:
            List of HealthSummary models (EnergyHealthSummary, HRHealthSummary, SleepHealthSummary, StepsHealthSummary)
            Mỗi model có field "type" để identify type_code
        """
        try:
            token = await self.login_and_get_token()
            if not token or not self.base_url:
                logger.warning(
                    "⚠️ Cannot get health summaries by range: not authenticated or base_url not configured"
                )
                return None

            headers = {"Authorization": f"Bearer {self._bearer_token}"}
            # Call endpoint không có type_code để lấy tất cả types
            url = f"{self.base_url}/api/health/summaries"

            params = {
                "userId": user_id,
                "startDate": start_date,
                "endDate": end_date,
                "timezone": timezone,
            }

            logger.info(
                f"📊 Fetching all health summaries from {start_date} to {end_date}"
            )

            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.get(url, headers=headers, params=params)

                if response.status_code == 200:
                    data = response.json()
                    if isinstance(data, list):
                        # Parse JSON thành Pydantic models
                        parsed_summaries = []
                        type_counts = {}

                        for item in data:
                            try:
                                item_type = item.get("type", "UNKNOWN")

                                # Parse based on type
                                if item_type == "ENERGY":
                                    summary = EnergyHealthSummary.model_validate(item)
                                elif item_type == "HR":
                                    summary = HRHealthSummary.model_validate(item)
                                elif item_type == "SLEEP":
                                    summary = SleepHealthSummary.model_validate(item)
                                elif item_type == "STEPS":
                                    summary = StepsHealthSummary.model_validate(item)
                                else:
                                    logger.warning(
                                        f"⚠️ Unknown health summary type: {item_type}, skipping"
                                    )
                                    continue

                                parsed_summaries.append(summary)
                                type_counts[item_type] = (
                                    type_counts.get(item_type, 0) + 1
                                )

                            except Exception as parse_error:
                                logger.warning(
                                    f"⚠️ Failed to parse health summary: {str(parse_error)}, skipping item"
                                )
                                continue

                        logger.info(
                            f"✅ Successfully retrieved and parsed {len(parsed_summaries)} health summaries: {', '.join([f'{k}: {v}' for k, v in type_counts.items()])}"
                        )
                        return parsed_summaries if parsed_summaries else None
                    else:
                        logger.warning(f"⚠️ Unexpected response format: {type(data)}")
                        return None
                else:
                    logger.warning(
                        f"⚠️ Failed to get health summaries by range: {response.status_code}"
                    )
                    return None

        except Exception as e:
            logger.error(f"❌ Error getting health summaries by range: {str(e)}")
            return None

    async def get_health_summary_latest_by_type(
        self,
        summary_type: str,
        user_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        try:
            if not await self._ensure_authenticated():
                return None

            params: Dict[str, Any] = {}
            if user_id:
                params["userId"] = user_id

            url = f"{self.base_url}/api/health/summaries/{summary_type}/latest"
            logger.info(f"📡 GET {url}")

            response = await self._make_request("GET", url, params=params)
            if not response:
                return None

            data = response.json()
            return data if isinstance(data, dict) else None
        except Exception as e:
            logger.error(f"❌ Error getting latest health summary for {summary_type}: {str(e)}")
            return None

    async def get_health_stats_daily(
        self,
        sample_type: str,
        start_date: str,
        end_date: str,
        user_id: Optional[str] = None,
        timezone: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        try:
            if not await self._ensure_authenticated():
                return []

            params: Dict[str, Any] = {
                "startDate": start_date,
                "endDate": end_date,
            }
            if user_id:
                params["userId"] = user_id
            if timezone:
                params["timezone"] = timezone

            url = f"{self.base_url}/api/health/stats/{sample_type}/daily"
            logger.info(f"📡 GET {url}")

            response = await self._make_request("GET", url, params=params)
            if not response:
                return []

            data = response.json()
            return data if isinstance(data, list) else []
        except Exception as e:
            logger.error(f"❌ Error getting daily health stats for {sample_type}: {str(e)}")
            return []

    async def get_sleep_stages_history(
        self,
        user_id: str,
        start_date: str,
        end_date: str,
        timezone: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Fetch multi-day sleep stage breakdown (deep / rem / core) for a date range."""
        try:
            summaries = await self.get_health_summaries_by_range(
                user_id, start_date, end_date, timezone
            )
            if not summaries:
                return []
            stages: List[Dict[str, Any]] = []
            for s in summaries:
                if getattr(s, "type", None) == "SLEEP" and hasattr(s, "data"):
                    for entry in s.data:
                        stages.append({
                            "date": entry.date,
                            "deep_min": int(entry.deep * 60),
                            "rem_min": int(entry.rem * 60),
                            "light_min": int(entry.core * 60),
                        })
            return stages
        except Exception as e:
            logger.error(f"❌ Error getting sleep stages history: {str(e)}")
            return []

    async def get_finance_summary(
        self, user_id: str, month: str
    ) -> Optional[Dict[str, Any]]:
        """Get finance summary for a month."""
        try:
            if not await self._ensure_authenticated():
                return None
            url = f"{self.base_url}/api/finance/summary"
            params = {"userId": user_id, "month": month}
            response = await self._make_request("GET", url, params=params)
            if not response:
                return None
            data = response.json()
            return data if isinstance(data, dict) else None
        except Exception as e:
            logger.error(f"❌ Error getting finance summary: {str(e)}")
            return None

    async def get_balance_score(
        self,
        user_id: str,
        date: str,
        end_date: Optional[str] = None,
        timezone: Optional[str] = None,
    ) -> Optional[BalanceScoreDTO]:
        """Get balance score for a specific date or date range."""
        try:
            if not await self._ensure_authenticated():
                return None

            # If end_date is provided, use range endpoint
            if end_date:
                scores = await self.get_balance_scores_by_range(
                    user_id=user_id,
                    start_date=date,
                    end_date=end_date,
                    timezone=timezone,
                )
                # Return the latest score from the range
                if scores:
                    return scores[-1] if scores else None
                return None

            url = f"{self.base_url}/api/balance/score"
            params: Dict[str, Any] = {"userId": user_id, "date": date}
            if timezone:
                params["timezone"] = timezone
            response = await self._make_request("GET", url, params=params)
            if not response:
                return None
            data = response.json()
            if not isinstance(data, dict):
                return None
            return BalanceScoreDTO.model_validate(data)
        except Exception as e:
            logger.error(f"❌ Error getting balance score: {str(e)}")
            return None

    async def get_balance_scores_by_range(
        self,
        user_id: str,
        start_date: str,
        end_date: str,
        timezone: Optional[str] = None,
    ) -> List[BalanceScoreDTO]:
        """Get balance scores by range from /api/balance."""
        try:
            if not await self._ensure_authenticated():
                return []
            url = f"{self.base_url}/api/balance"
            params: Dict[str, Any] = {
                "userId": user_id,
                "startDate": start_date,
                "endDate": end_date,
            }
            if timezone:
                params["timezone"] = timezone
            response = await self._make_request("GET", url, params=params)
            if not response:
                logger.warning(f"balance_scores_by_range: empty response from {url}")
                return []
            data = response.json()
            if not isinstance(data, list):
                logger.warning(f"balance_scores_by_range: response not list, type={type(data).__name__}")
                return []
            logger.info(
                f"balance_scores_by_range: raw items={len(data)}, "
                f"sample={data[:2] if data else '[]'}"
            )
            parsed: List[BalanceScoreDTO] = []
            for item in data:
                if not isinstance(item, dict):
                    continue
                try:
                    parsed.append(BalanceScoreDTO.model_validate(item))
                except Exception:
                    continue
            # Return plain dicts so downstream collectors can use dict.get(...)
            # uniformly (some callers iterate with isinstance(s, dict)).
            return [p.model_dump() for p in parsed]
        except Exception as e:
            logger.error(f"❌ Error getting balance scores by range: {str(e)}")
            return []

    async def get_balance_streak(
        self,
        user_id: Optional[str] = None,
        timezone: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        try:
            if not await self._ensure_authenticated():
                return None

            params: Dict[str, Any] = {}
            if user_id:
                params["userId"] = user_id
            if timezone:
                params["timezone"] = timezone

            url = f"{self.base_url}/api/balance/streak"
            logger.info(f"Fetching balance streak from: {url}")
            response = await self._make_request("GET", url, params=params)

            if not response:
                return None
            data = response.json()
            return data if isinstance(data, dict) else None
        except Exception as e:
            logger.error(f"❌ Error getting balance streak: {str(e)}")
            return None

    async def get_finance_goals(self, user_id: str) -> List[Dict[str, Any]]:
        """Get finance goals."""
        try:
            if not await self._ensure_authenticated():
                return []
            url = f"{self.base_url}/api/finance/goals"
            params = {"userId": user_id}
            response = await self._make_request("GET", url, params=params)
            if not response:
                return []
            data = response.json()
            return data if isinstance(data, list) else []
        except Exception as e:
            logger.error(f"❌ Error getting finance goals: {str(e)}")
            return []

    async def get_finance_bills(self, user_id: str) -> List[Dict[str, Any]]:
        """Get finance bills."""
        try:
            if not await self._ensure_authenticated():
                return []
            url = f"{self.base_url}/api/finance/bills"
            params = {"userId": user_id}
            response = await self._make_request("GET", url, params=params)
            if not response:
                return []
            data = response.json()
            return data if isinstance(data, list) else []
        except Exception as e:
            logger.error(f"❌ Error getting finance bills: {str(e)}")
            return []

    async def get_finance_logs(
        self,
        user_id: str,
        month: Optional[str] = None,
        log_type: Optional[str] = None,
        categories: Optional[List[str]] = None,
        page: int = 1,
        size: int = 100,
    ) -> List[Dict[str, Any]]:
        try:
            if not await self._ensure_authenticated():
                return []
            url = f"{self.base_url}/api/finance/logs"
            params: Dict[str, Any] = {
                "userId": user_id,
                "page": page,
                "size": size,
            }
            if month:
                params["month"] = month
            if log_type:
                params["logType"] = log_type
            if categories:
                params["categories"] = categories
            response = await self._make_request("GET", url, params=params)
            if not response:
                return []
            data = response.json()
            if isinstance(data, dict):
                content = data.get("content", [])
                return content if isinstance(content, list) else []
            return data if isinstance(data, list) else []
        except Exception as e:
            logger.error(f"❌ Error getting finance logs: {str(e)}")
            return []

    async def get_finance_budgets(
        self,
        user_id: str,
        budget_type: Optional[str] = None,
        categories: Optional[List[str]] = None,
        active: bool = True,
    ) -> List[Dict[str, Any]]:
        try:
            if not await self._ensure_authenticated():
                return []
            url = f"{self.base_url}/api/finance/budgets"
            params: Dict[str, Any] = {"userId": user_id, "active": str(active).lower()}
            if budget_type:
                params["budgetType"] = budget_type
            if categories:
                params["categories"] = categories
            response = await self._make_request("GET", url, params=params)
            if not response:
                return []
            data = response.json()
            return data if isinstance(data, list) else []
        except Exception as e:
            logger.error(f"❌ Error getting finance budgets: {str(e)}")
            return []

    # ─────────────────────────────────────────────────────────────
    # BE Snapshot Pipeline API (Monthly Insight)
    # ─────────────────────────────────────────────────────────────
    # Production endpoint:
    #     GET {base}/api/pipeline/daily-user-snapshots
    #         ?userId=<id>&startDate=YYYY-MM-DD&endDate=YYYY-MM-DD
    #         &page=<n>&size=<n>
    # Returns Spring Pageable shape: {content: [...], totalPages, last, ...}.
    # Auth: bearer token obtained via login_and_get_token() using BE_APP_EMAIL +
    #       BE_APP_PASSWORD env credentials (same as every other API in this
    #       service — token cached + auto-refreshed).

    _SNAPSHOT_PAGE_SIZE: int = 100  # > 31-day month → typically 1 page
    _SNAPSHOT_MAX_PAGES: int = 20  # safety cap on pagination loop

    async def get_daily_snapshots(
        self,
        profile_id: str,
        start: date,
        end: date,
        timezone: Optional[str] = None,
        language: Optional[str] = None,
    ) -> List[DailySnapshot]:
        """Fetch daily snapshot rows from BE pipeline for ``[start, end]`` inclusive.

        Endpoint::

            GET {BE_APP_INTEGRATE_HOST}/api/pipeline/daily-user-snapshots
                ?userId=<id>&startDate=YYYY-MM-DD&endDate=YYYY-MM-DD&page=N&size=N

        ``timezone`` / ``language`` are accepted for interface symmetry but the
        production pipeline endpoint does not use them.

        Auth: uses ``_ensure_authenticated()`` which runs ``login_and_get_token``
        with BE_APP_EMAIL + BE_APP_PASSWORD env credentials. Token is cached and
        auto-refreshed (same pattern as every other API in this service).
        """
        if not await self._ensure_authenticated():
            return []

        # Log full token for debugging (per user request)
        logger.info(f"🔑 Snapshot API")

        url = f"{self.base_url}/api/pipeline/daily-user-snapshots"
        base_params: Dict[str, Any] = {
            "userId": profile_id,
            "startDate": start.isoformat(),
            "endDate": end.isoformat(),
            "size": self._SNAPSHOT_PAGE_SIZE,
        }

        all_items: List[Dict[str, Any]] = []
        for page_num in range(1, self._SNAPSHOT_MAX_PAGES + 1):
            params = {**base_params, "page": page_num}
            query_str = "&".join(f"{k}={v}" for k, v in params.items())
            logger.info(f"📡 GET {url}?{query_str}")
            response = await self._make_request("GET", url, params=params)
            if response is None:
                raise RuntimeError(
                    f"Snapshot API request failed (page {page_num}); "
                    "see logs for HTTP status and response body."
                )
            try:
                payload = response.json()
            except ValueError as e:
                raise RuntimeError(f"Snapshot API returned non-JSON: {e}") from e

            items = payload.get("content") if isinstance(payload, dict) else None
            if items is None and isinstance(payload, dict):
                # Backwards compat with old ``{"data": [...]}`` shape
                items = payload.get("data")
            if items is None and isinstance(payload, list):
                items = payload
            if not isinstance(items, list):
                raise RuntimeError(
                    f"Snapshot API: expected list/content, got {type(items).__name__}"
                )
            all_items.extend(items)

            is_last = payload.get("last", True) if isinstance(payload, dict) else True
            if is_last or not items:
                break

        snapshots: List[DailySnapshot] = []
        for idx, item in enumerate(all_items):
            try:
                snapshots.append(DailySnapshot.model_validate(item))
            except Exception as e:
                logger.warning(f"⚠️ Skipping daily snapshot #{idx} (parse failed): {e}")
        logger.info(
            f"✅ Snapshot API returned {len(snapshots)} valid daily rows "
            f"for user {profile_id}"
        )
        return snapshots